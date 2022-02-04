// for chip button selection
$(document).ready(function(){
        $('.chip').click(function() {
            $('.chip').not(this).prop('checked', false);
        });
    });
   

$('#form').on('submit', function (e) {
    if ($("input[type=radio]:checked").length === 0) {
        e.preventDefault();
        alert('no way you submit it without selecting a model');
        return false;
    }
  });

// for select option
  var buttonCommon = {
    exportOptions: {
        columns:  [ 0, 2],
      
        format: {
            body: function (data, row, columns, node) {
                // if it is select
                
                if (columns == 1) {
                    return $(node).find("option:selected").text()
                } else return data
            }
        }
    }
};

let tableArray = [];   
$(document).ready(function() {
const table = $('#proxies').DataTable( {
dom: 'Bfrtip',
buttons: [
        'copy', $.extend(true, {}, buttonCommon, {
            extend: "csv"
        })
        ],
        select: {
style: "multi",
selector: "td:first-child",
CSV: "CSV"
},

} );

// seleced rows in array
table.on("select.dt", function() {
tableArray = table
.rows(".selected")
.data()
.toArray();
$("#output").html(tableArray);
});
// Deselect from Array
table.on("deselect.dt", function() {
tableArray = table
.rows(".selected")
.data()
.toArray();
$("#output").html(tableArray);
});


} );
